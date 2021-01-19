import numpy as np
import cv2
import glob
import os

def detect_nails3(filepath,image,DEBUG):
    name = os.path.basename(filepath)
    if DEBUG:
        print('\n====== NAILS 3 ======')
        cv2.imshow("Starting image", image)

    # Apply special HSV filter to remove everything but nails
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([40, 0, 0], dtype = "uint8")
    upper = np.array([179, 66, 255], dtype = "uint8")
    mask = cv2.inRange(hsvim, lower, upper)
    mask = cv2.medianBlur(mask,3)
    if DEBUG:
        cv2.imshow("HSV filter", mask)
        cv2.imwrite('./example/hsv4-' + name, mask)
    
    # Morphology operations to remove noises and enlarge area of nails
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask,3)
    if DEBUG:
        cv2.imshow("Morphology", mask)
        cv2.imwrite('./example/close4-' + name, mask)

    # Find nails
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = image.shape[0]*image.shape[1]
    nails = []
    minRatio = 0.004
    maxRatio = 0.028
    if area > 300000:
        minRatio = 0.0017
        maxRatio = 0.02

    # Find nails using size condition
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > minRatio*area and cv2.contourArea(contours[i]) <= maxRatio*area:
            nails.append(contours[i])

     # Draw found contours (nails) and bounding boxes for them
    boxes = []
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for i in range(len(nails)):
        rect = cv2.minAreaRect(nails[i])
        box = np.int0(cv2.boxPoints(rect))
        boxes.append(box)
        cv2.drawContours(mask, [nails[i]], 0, (255,255,255), cv2.FILLED)

    image_box = cv2.imread(filepath)
    for i in range(len(boxes)):
        cv2.drawContours(image_box, [boxes[i]], -1, (0, 0, 255), 2)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imshow("final mask", mask)
        cv2.imshow("box",image_box)
        cv2.imwrite('./example/Mask4-' + name, mask)
        cv2.imwrite('./example/Box4-' + name, image_box)
        cv2.waitKey(0); cv2.destroyAllWindows();

    return mask, image_box