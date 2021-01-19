import numpy as np
import cv2
import glob
import os
from nails2 import detect_nails2

def detect_nails(filepath,image,DEBUG):
    name = os.path.basename(filepath)
    if DEBUG:
        print('\n ======= NAILS ======')
        cv2.imshow("Starting image", image)

    # Apply HSV filter to separate nails
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([116, 0, 0], dtype = "uint8")
    upper = np.array([179, 130, 255], dtype = "uint8")
    mask = cv2.inRange(hsvim, lower, upper)
    mask = cv2.medianBlur(mask,5)
    if DEBUG:
        cv2.imshow("HSV filter 2", mask)
        cv2.imwrite('./example/hsv2-' + name, mask)

    # Morphology operations to remove background noises and close contours
    kernel = np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if DEBUG:
        cv2.imshow("Morphology", mask)
        cv2.imwrite('./example/close2-' + name, mask)
    
    # Dilate the obtained mask to enlarge area of nails
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask,3)
    if DEBUG:
        cv2.imshow("Dilate", mask)
        cv2.imwrite('./example/dilate-' + name, mask)
        cv2.waitKey(0); cv2.destroyAllWindows();

    # Prepare variables and find nails using size condition
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = image.shape[0]*image.shape[1]
    nails = []
    if area > 500000:
        minRatio = 0.0007
        maxRatio = 0.0052
    if 465500 < area <= 500000:
        minRatio = 0.0014
        maxRatio = 0.012
    else:
        minRatio = 0.0014
        maxRatio = 0.028
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > minRatio*area and cv2.contourArea(contours[i]) <= maxRatio*area:
            nails.append(contours[i])
    
    # If there are too few nails, apply different HSV filter
    if len(nails) < 3 or (sum(cv2.contourArea(x) < 0.00177*area for x in nails) >= 2 and area < 60000) or len(nails) > 10:
        mask,image_box = detect_nails2(filepath,image, DEBUG)
        return mask, image_box 
    
    # Draw obtained contours on black mask
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    boxes = []
    for i in range(len(nails)):
        rect = cv2.minAreaRect(nails[i])
        box = np.int0(cv2.boxPoints(rect))
        boxes.append(box)
        cv2.drawContours(mask, [nails[i]], 0, (255,255,255), cv2.FILLED)
            
    # Draw bounding boxes on original image
    image_box = cv2.imread(filepath)
    for i in range(len(boxes)):
        cv2.drawContours(image_box, [boxes[i]], -1, (0, 0, 255), 2)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imshow("Final mask", mask)
        cv2.imshow("Boxes",image_box)
        cv2.imwrite('./example/Mask2-' + name, mask)
        cv2.imwrite('./example/Box2-' + name, image_box)
        cv2.waitKey(0); cv2.destroyAllWindows();

    return mask,image_box


