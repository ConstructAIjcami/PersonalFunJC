import numpy as np
import cv2

image_path ="data/objgap/gooddrywall1.jpg"
image = cv2.imread(image_path)
original = image
result = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0,0,0], dtype="uint8")
upper = np.array([180,255,80], dtype="uint8")
mask = cv2.inRange(image, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

area = 0

for c in cnts:
    area += cv2.contourArea(c)
    cv2.drawContours(result,[c], 0, (0,0,255), 2)
print(area)
if area > 300:
    print("Issue detected")
    #cv2.imshow('mask', mask)
    cv2.putText(result, "Issue detected", (30,185),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Result",result)
else:
    print("No Issue detected")
    cv2.putText(original, "No Issue detected", (30,185),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Result",original)
    #cv2.imshow('mask', mask)
cv2.waitKey(0)