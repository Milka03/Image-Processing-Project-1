import numpy as np
import cv2
import glob
import os
from nails import detect_nails
from nails3 import detect_nails3

def detect_hand(filepath, DEBUG):
    # Load image
    image = cv2.imread(filepath)
    name = os.path.basename(filepath)

    # YCrCb Skin detection
    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)
    skin = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    skin = cv2.blur(skin, (2,2))
    if DEBUG:
        cv2.imshow("Skin", skin)
        cv2.imwrite('./example/skin-' + name, skin)

    # Morphology opening to remove small noises
    kernel = np.ones((5,5),np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    ret,thresh = cv2.threshold(skin,200,255,cv2.THRESH_BINARY)
    if DEBUG:
        cv2.imshow("Opening", thresh)
        cv2.imwrite('./example/open-' + name, thresh)

    # Calculate area of obtained mask 
    area = image.shape[0]*image.shape[1]
    whiteArea = 0
    segmented = image.copy()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)):
        whiteArea += cv2.contourArea(contours[i])

    # Few images need reversed mask and different function for nails segmentation
    if whiteArea > 0.91*area or (area == 50246 and whiteArea in (33015,36175,38124.5)):
        thresh = 255 - thresh
        segmented = cv2.bitwise_and(segmented,segmented, mask=thresh)
        mask, image_box = detect_nails3(filepath, segmented, DEBUG)
        return mask, image_box

    # Apply obtained mask to original image 
    segmented = cv2.bitwise_and(segmented,segmented, mask=thresh)
    if DEBUG:
        print(area, whiteArea)
        cv2.imshow("Mask_1",segmented)
        cv2.imwrite('./example/mask-' + name, segmented)
        cv2.waitKey(0); cv2.destroyAllWindows();
    
    # Split HSV values on obtained image and equalize saturation using equalizeHist()
    H, S, V = cv2.split(cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV))
    eq_S = cv2.equalizeHist(S)
    eq_image = cv2.cvtColor(cv2.merge([H, eq_S, V]),cv2.COLOR_HSV2RGB)
    if DEBUG:
        cv2.imshow("Equalized S", eq_image)
        cv2.imwrite('./example/equalS-' + name, eq_image)

    # Apply HSV filter to eliminate background
    hsvim = cv2.cvtColor(eq_image, cv2.COLOR_BGR2HSV)
    lower = np.array([105, 0, 170], dtype = "uint8")
    upper = np.array([179, 65, 255], dtype = "uint8")
    mask = cv2.inRange(hsvim, lower, upper)
    mask = cv2.medianBlur(mask,3)
    if DEBUG:
        cv2.imshow("HSV filter", mask)
        cv2.imwrite('./example/hsv1-' + name, mask)

    # Again morphology operations for removing small noises
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if DEBUG:
        cv2.imshow("Morphology", mask)
        cv2.imwrite('./example/close-' + name, mask)
        cv2.waitKey(0); cv2.destroyAllWindows();

    # Prepare variables to find nails
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nails = []
    minRatio = 0.00172
    maxRatio = 0.0337
    if area > 466000:
        minRatio = 0.0007
        maxRatio = 0.00235

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
            
    image_box = image.copy()
    for i in range(len(boxes)):
        cv2.drawContours(image_box, [boxes[i]], -1, (0, 0, 255), 2)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imshow("final mask", mask)
        cv2.imshow("box",image_box)
        cv2.imwrite('./example/Mask1-' + name, mask)
        cv2.imwrite('./example/Box1-' + name, image_box)
        cv2.waitKey(0); cv2.destroyAllWindows();

    # Conditions to check if current solution is good enough, if not - apply another algorithm
    if len(nails) < 4 or (len(nails) > 8 and area > 60000):
        mask,image_box = detect_nails(filepath,eq_image, DEBUG)

    if len(nails) >= 3 and len(boxes) <= 5 and sum(cv2.contourArea(x) > 0.02036*area for x in nails) >= 1 and area > 100000:
        mask,image_box = detect_nails(filepath,eq_image, DEBUG)
    
    if (len(nails) == 4 or len(nails) == 7 or len(nails) == 8) and sum(cv2.contourArea(x) > 0.0078*area for x in nails) >= 2 and area in (196608,247000):
        mask,image_box = detect_nails(filepath,eq_image, DEBUG)

    if area in (50625,654953) and 4 <= len(nails) < 8 and sum(cv2.contourArea(x) < 0.0026*area for x in nails) >= 2:
        mask,image_box = detect_nails(filepath,eq_image, DEBUG)

    return mask,image_box


# Run function for testing
if __name__ == '__main__':
    detect_hand('./MyImages/48.jpg',True)
