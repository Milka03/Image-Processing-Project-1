# Image-Processing-Project-1
Segmentation of fingernails using OpenCV

The object of this project is to segment pictures from "images" folder to obtain masks from "labels" folder (target masks).\
By invoking main_function.py the algorithm iterates through all images in "images" folder and uses appropriate function to separate fingernails and get the mask.
All masks are saved in Results/masks folder and additionally bounding boxes for found nails are drawn on copy of the original image and saved in Results/boxes folder.

The accuracy of obtained masks is calculated using Intersection over Union (IoU) and Dice coefficient. The IoU and Dice for each image as well as mean result for whole dataset are saved in file result_masks.txt.
