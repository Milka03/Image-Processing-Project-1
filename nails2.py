import numpy as np
import cv2
import glob
import os


def detect_nails2(filepath,image,DEBUG):
    name = os.path.basename(filepath)
    if DEBUG:
        print('\n ======= NAILS 2 =======')
        cv2.imshow("Starting image", image)

    # Apply HSV filter to filter out nails
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([112, 0, 0], dtype = "uint8")
    upper = np.array([179, 130, 255], dtype = "uint8")
    mask = cv2.inRange(hsvim, lower, upper)
    mask = cv2.medianBlur(mask,5)
    if DEBUG:
        cv2.imshow("HSV filter 3", mask)
        cv2.imwrite('./example/hsv3-' + name, mask)
    
    # Morphology operations to remove noises
    kernel = np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if DEBUG:
        cv2.imshow("Morphology", mask)
        cv2.imwrite('./example/close3-' + name, mask)
    
    # Small dilation to enlarge area of nails
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask,3)
    if DEBUG:
        cv2.imshow("Dilate", mask)
        cv2.imwrite('./example/dilate2-' + name, mask)
        cv2.waitKey(0); cv2.destroyAllWindows();

    # Find estimated nails using size condition
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = image.shape[0]*image.shape[1]
    nails = []
    if area > 466000:
        minRatio = 0.0007
        maxRatio = 0.0176
    else:
        minRatio = 0.0014
        maxRatio = 0.0187
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > minRatio*area and cv2.contourArea(contours[i]) <= maxRatio*area:
            nails.append(contours[i])

    # Draw nails' contours to obtain final mask
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    boxes = []
    for i in range(len(nails)):
        rect = cv2.minAreaRect(nails[i])
        box = np.int0(cv2.boxPoints(rect))
        boxes.append(box)
        cv2.drawContours(mask, [nails[i]], 0, (255,255,255), cv2.FILLED)

    # Draw bounding boxes
    image_box = cv2.imread(filepath)
    for i in range(len(boxes)):
        cv2.drawContours(image_box, [boxes[i]], -1, (0, 0, 255), 2)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imshow("Final_Mask", mask)
        cv2.imshow("Boxes",image_box)
        cv2.imwrite('./example/Mask3-' + name, mask)
        cv2.imwrite('./example/Box3-' + name, image_box)
        cv2.waitKey(0); cv2.destroyAllWindows();

    return mask, image_box
